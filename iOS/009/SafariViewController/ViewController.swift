//
//  ViewController.swift
//  SafariViewController
//
//  Created by Satyadev on 09/08/17.
//  Copyright © 2017 Satyadev Chauhan. All rights reserved.
//

import UIKit
import SafariServices

class ViewController: UIViewController {
    
    private var urlString:String = "https://google.com"
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    @IBAction func openInSafari_Clicked(_ sender: Any) {
        print("openInSafari_Clicked:\(sender)")
        
        if let url = URL(string: self.urlString) {
            
            UIApplication.shared.open(url, options: [:], completionHandler: { (success) in
                
                if success {
                    print("open:\(url)completionHandler:\(success)")
                }
                
            })
        }
    }
    
    @IBAction func openInWebview_Clicked(_ sender: Any) {
        print("openInWebview_Clicked:\(sender)")
        
        let webViewController = WebviewController(nibName: "WebviewController", bundle: nil)
        webViewController.url = URL(string:urlString)
        
        self.present(webViewController, animated: true, completion: nil)
    }
    
    @IBAction func openInWKWebview_Clicked(_ sender: Any) {
        print("openInWKWebview_Clicked:\(sender)")
        
        let storyboard = UIStoryboard(name: "WK", bundle: nil)
        
        /*
         * Storyboard initial view has navigation controller
         */
         /*if let controller = storyboard.instantiateInitialViewController() {
            self.present(controller, animated: true, completion: nil)
         }*/
        
        /*
         * Storyboard WKWebviewController initialising with controller
         */
        let wkViewController = storyboard.instantiateViewController(withIdentifier: "WKWebviewController") as! WKWebviewController
        wkViewController.url = URL(string:urlString)
        
        // Creating a navigation controller with viewController at the root of the navigation stack.
        let navController = UINavigationController(rootViewController: wkViewController)
        self.present(navController, animated: true, completion: nil)
        
    }
    
    @IBAction func openInSafariViewController_Clicked(_ sender: Any) {
        print("openInSafariViewController_Clicked:\(sender)")
        
        if let url = URL(string: self.urlString) {
            
            let svc = SFSafariViewController.init(url: url)
            svc.delegate = self
            self.present(svc, animated: true, completion: nil)
            
        }
    }
}

//MARK: SFSafariViewControllerDelegate
extension ViewController: SFSafariViewControllerDelegate {
    
    /*! @abstract Called when the view controller is about to show UIActivityViewController after the user taps the action button.
     @param URL the URL of the web page.
     @param title the title of the web page.
     @result Returns an array of UIActivity instances that will be appended to UIActivityViewController.
     */
    public func safariViewController(_ controller: SFSafariViewController, activityItemsFor URL: URL, title: String?) -> [UIActivity] {
        
        print("safariViewController:controller:URL:\(URL.absoluteString)")
        return []
    }
    
    /*! @abstract Delegate callback called when the user taps the Done button. Upon this call, the view controller is dismissed modally. */
    public func safariViewControllerDidFinish(_ controller: SFSafariViewController) {
        print("safariViewControllerDidFinish")
    }
    
    /*! @abstract Invoked when the initial URL load is complete.
     @param success YES if loading completed successfully, NO if loading failed.
     @discussion This method is invoked when SFSafariViewController completes the loading of the URL that you pass
     to its initializer. It is not invoked for any subsequent page loads in the same SFSafariViewController instance.
     */
    public func safariViewController(_ controller: SFSafariViewController, didCompleteInitialLoad didLoadSuccessfully: Bool){
        print("safariViewController:controller:didLoadSuccessfully:\(didLoadSuccessfully)")
    }
}

