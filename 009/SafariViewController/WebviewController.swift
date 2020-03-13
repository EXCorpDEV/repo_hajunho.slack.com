//
//  WebviewController.swift
//  SafariViewController
//
//  Created by Satyadev on 10/08/17.
//  Copyright © 2017 Satyadev Chauhan. All rights reserved.
//

import UIKit

class WebviewController: UIViewController {

    var url: URL!
    
    @IBOutlet weak var webView: UIWebView!
    
    @IBOutlet weak var barView: UIView!
    @IBOutlet weak var urlField: UITextField!
    
    @IBOutlet weak var backButton: UIBarButtonItem!
    @IBOutlet weak var forwardButton: UIBarButtonItem!
    @IBOutlet weak var reloadButton: UIBarButtonItem!
    
    @IBOutlet weak var progressView: UIProgressView!
    
    let refreshControl = UIRefreshControl()
    
    override func viewDidLoad() {
        super.viewDidLoad()

        // Do any additional setup after loading the view.
        barView.frame = CGRect(x:0, y: 0, width: view.frame.width, height: 30)
        
        //Refresh Control
        refreshControl.addTarget(self, action: #selector(self.reload(_:)), for: UIControl.Event.valueChanged)
        webView.scrollView.addSubview(refreshControl)
        webView.delegate = self;
        
        if !url.absoluteString.isEmpty {
            loadURL(url)
        }
        
        backButton.isEnabled = false
        forwardButton.isEnabled = false
        
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

    override func viewWillTransition(to size: CGSize, with coordinator: UIViewControllerTransitionCoordinator) {
        barView.frame = CGRect(x:0, y: 0, width: size.width, height: 30)
    }

    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destinationViewController.
        // Pass the selected object to the new view controller.
    }
    */
    
    func loadURL(_ url: URL) {
        
        urlField.text = url.absoluteString
        
        let request = URLRequest(url:url)
        webView.loadRequest(request)
    }
    
    func updateToolBar() {
        forwardButton.isEnabled = webView.canGoForward;
        backButton.isEnabled = webView.canGoBack;
    }
}

//MARK: UITextFieldDelegate
extension WebviewController: UITextFieldDelegate {
    
    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        
        urlField.resignFirstResponder()
        let request = URLRequest(url:URL(string: urlField.text!)!)
        webView.loadRequest(request)
        
        return false
    }
}

//MARK: Actions
extension WebviewController {
    
    @IBAction func done(_ sender: Any) {
        dismiss(animated: true, completion: nil)
    }
    
    @IBAction func stop(_ sender: Any) {
        webView.stopLoading()
        progressView.setProgress(0, animated: true);
        UIApplication.shared.isNetworkActivityIndicatorVisible = false
        
        urlField.selectAll(urlField)
        urlField.becomeFirstResponder()
    }
    
    @IBAction func back(_ sender: UIBarButtonItem) {
        webView.goBack()
        urlField.text = webView.request?.url?.absoluteString
    }
    
    @IBAction func forward(_ sender: UIBarButtonItem) {
        webView.goForward()
        urlField.text = webView.request?.url?.absoluteString
    }
    
    @IBAction func reload(_ sender: UIBarButtonItem) {
        webView.reload()
    }
}

extension WebviewController: UIWebViewDelegate {
    
    public func webView(_ webView: UIWebView, shouldStartLoadWith request: URLRequest, navigationType: UIWebView.NavigationType) -> Bool {
        
        print("URL:\(request.url?.absoluteString ?? "")")
        
        return true
    }
    
    func webViewDidStartLoad(_ webView: UIWebView) {
        UIApplication.shared.isNetworkActivityIndicatorVisible = true
        updateToolBar()
    }
    
    func webViewDidFinishLoad(_ webView: UIWebView) {
        UIApplication.shared.isNetworkActivityIndicatorVisible = false
        updateToolBar()
        refreshControl.endRefreshing()
        //self.navigationTitle.title = webView.stringByEvaluatingJavaScriptFromString("document.title")
    }
    
    func webView(_ webView: UIWebView, didFailLoadWithError error: Error) {
        updateToolBar()
        refreshControl.endRefreshing()
        UIApplication.shared.isNetworkActivityIndicatorVisible = false
    }
}
