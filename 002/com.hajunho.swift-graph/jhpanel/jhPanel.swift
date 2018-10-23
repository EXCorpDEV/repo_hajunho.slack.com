//
//  jhPanel.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 15..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

class jhPanel : jhDraw, jhPanel_p, observer_p {
    
    var jhEnforcingMode: Bool = false
    var jhPanelID: Int = 0
    var dataLayer : CALayer = CALayer(layer: 0)
    var axisLayer : CALayer = CALayer(layer: 0)
    
    internal var mContext : CGContext? = nil
    
    //stored property realted with DATAs
    internal var mCountOfDatas : Int = 0
    internal var mMaxValueOfDatas : CGFloat = 0
    internal var mMinvalueOfDatas : CGFloat = 0
    
    //stored property related with Drawing
    internal let mFixedPanelWidth : CGFloat = jhDraw.maxR //basic ratio 0~10000.0
    internal let mFixedPanelHeight : CGFloat = jhDraw.maxR  //basic ratio
    
    //    private var mMargin : CGFloat = 1333.3 //1000.0 is 13.3..%, margin between panel & graph area 0<=martgin<10000.0
    internal var mMargin : CGFloat = 300 //1000.0 is 13.3..%, margin between panel & graph area 0<=martgin<10000.0
    
    internal var mPanelWidth : CGFloat? = nil
    internal var mPanelHeight : CGFloat? = nil
    
    internal var mLineWidth : CGFloat = 1
    internal var mColor : CGColor = UIColor.blue.cgColor
    
    
    /// Axes
    var mUnitOfHorizontalAxes : CGFloat = 100
    var mcountOfHorizontalAxes : Int = 3
    
    internal var mVerticalRatioToDraw_view : CGFloat = 1.0
    
    //calculated property related with DATAs' View
    internal var mAllofCountOfDatas : Int {
        get {
            return jhData.nonNetworkData.count
//            return jhData.mDatas[0]?.d.count ?? 0
        }
    }
    
    var axisDistance : CGFloat {
        get {
            return (jhDraw.maxR  - mMargin * 2) / CGFloat(jhData.mCountOfaxes_view+1)
        }
        //        set(distance) {
        //            jhData.mCountOfaxes_view = Int(jhDraw.maxR  / CGFloat(distance))
        //        }
    }
    
    override init(frame: CGRect) {
        if GS.shared.logLevel.contains(.graphPanel) { print("jhPanel override init(\(frame.width), \(frame.height))")}
        super.init(frame: frame)
        self.layer.isGeometryFlipped = true
        mContext = UIGraphicsGetCurrentContext()
        self.mPanelWidth = frame.width
        self.mPanelHeight = frame.height
        if GS.shared.logLevel.contains(.graphPanel) { print("jhPanel init color", mLineWidth)}
    }
    
    override func draw(_ rect: CGRect) {
        if GS.shared.logLevel.contains(.graphPanel) {
            print("jhPanel draw()")
        }
        
        self.mContext = UIGraphicsGetCurrentContext()
        
        drawPanel()
    }
    
    func drawPanel() {
        if GS.shared.logLevel.contains(.graphPanel) { print("drawPanel()") }
        
        initDatas()
        
        drawBackboard()
        drawDatas()
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    func changePanelSize(_ x : CGFloat, _ y : CGFloat) {
        self.mPanelWidth = x
        self.mPanelHeight = y
    }
    
    func getX(_ x: CGFloat) -> CGFloat? {
        var retX : CGFloat? = nil
        retX = x * mPanelWidth! / mFixedPanelWidth
        return retX
    }
    
    func getY(_ y: CGFloat) -> CGFloat? {
        var retY : CGFloat? = nil
        retY = y * mPanelHeight! / mFixedPanelHeight
        return retY
    }
    
    func drawLine(_ x1 : CGFloat, _ y1 : CGFloat, _ x2 : CGFloat, _ y2 : CGFloat) {
        if GS.shared.logLevel.contains(.graph) { print("panel_drawLine_\(x1), \(y1), \(x2), \(y2)")}
        jhDraw.worldLine(context: mContext, getX(x1)!, getY(y1)!, getX(x2)!, getY(y2)!, mLineWidth, mColor)
    }
    
    func drawLineWithColor(_ x1 : CGFloat, _ y1 : CGFloat, _ x2 : CGFloat, _ y2 : CGFloat, lineWidth : CGFloat, color : CGColor) {
        jhDraw.worldLine(context: mContext, getX(x1)!, getY(y1)!, getX(x2)!, getY(y2)!, lineWidth, color)
    }
    
    func drawRect(margin : CGFloat) {
        drawLine(margin, margin, mFixedPanelWidth-margin, margin)
        drawLine(mFixedPanelWidth-margin, margin, mFixedPanelWidth-margin, mFixedPanelHeight-margin)
        drawLine(mFixedPanelWidth-margin, mFixedPanelHeight-margin, margin, mFixedPanelHeight-margin)
        ////For DEBUG
        //        drawLine(0, 0, mFixedPanelWidth, mFixedPanelHeight)
        //        drawLine(0, mFixedPanelHeight, mFixedPanelWidth, 0)
        drawLine(margin, mFixedPanelHeight-margin, margin, margin)
    }
    
    func drawRect(margin : CGFloat, color : CGColor) {
        mColor = color
        drawRect(margin: margin)
    }
    
    /// draw X-axes, Y-axes
    func drawBackboard() {
        mColor = jhColor(r: 229, g: 229, b: 229, a: 1.0)
        drawRect(margin: mMargin)
        
        jhData.mCountOfaxes_view = mAllofCountOfDatas
        
        drawAxes()
    }
    
    func drawText(str : String, x : CGFloat, y : CGFloat, width : CGFloat, height : CGFloat) -> UIImageView {
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: width, height: height))
        let img = renderer.image { ctx in
            let paragraphStyle = NSMutableParagraphStyle()
            paragraphStyle.alignment = .center
            let attrs = [NSAttributedString.Key.font: UIFont(name: "".font1(), size: width/2)!, NSAttributedString.Key.paragraphStyle: paragraphStyle]
            let string = str
            string.draw(with: CGRect(x: 0, y: 0, width: width, height: 10), options: .usesLineFragmentOrigin, attributes: attrs, context: nil)
        }
        let imageView : UIImageView = UIImageView(frame: CGRect(x: getX(x)!, y: getY(y)!, width: width, height: height))
        imageView.image = img
        return imageView
    }
    
    func drawAxes() {
        
        axisLayer = jhDrawAxisLayer(axisDistance, mVerticalRatioToDraw_view, mMargin, mPanelWidth ?? 0, mPanelHeight ?? 0, mFixedPanelWidth, mFixedPanelHeight, layer: 0, panelID: 0)
        
        axisLayer.frame = CGRect(x: 0, y: 0, width: self.mPanelWidth!, height: self.mPanelHeight!) //TODO: will be changed.
        axisLayer.zPosition=1
        //        guideLine.isGeometryFlipped = true
        axisLayer.backgroundColor = UIColor(white: 1, alpha:0.5).cgColor
        self.layer.addSublayer(axisLayer)
        axisLayer.setNeedsDisplay()
        //        jhData.attachObserver(observer: self)
    }
    
    func initDatas() {
        
        var dataSource = getArrayOfData() {
            didSet {
            }
        }
        
        var maxValue : CGFloat = 0.0
        var minValue : CGFloat = jhDraw.maxR
        
        for element in dataSource {
            let _element = element as! NSArray
            let vDate = _element[0] as! CFDate
            let vNumber = _element[1] as! CGFloat
            
            if GS.shared.logLevel.contains(.graph) {
                print("datasrc2 \(vDate) \(vNumber)")
            }
            
            if vNumber > maxValue { maxValue = vNumber }
            if vNumber < minValue { minValue = maxValue }
            
            jhData.nonNetworkData.append(vNumber)
        }
        
        self.mMaxValueOfDatas = maxValue
        self.mMinvalueOfDatas = minValue
        
        self.mVerticalRatioToDraw_view = (jhDraw.maxR - (2*mMargin)) / self.mMaxValueOfDatas
        if GS.shared.logLevel.contains(.graph) {
            print("mVerticalRatioToDraw_view =", mVerticalRatioToDraw_view)
        }
    }
    
    func jhReSize(size : CGSize) {
        //        self.jhSceneFrameHeight = size.width
        //        self.jhSceneFrameHeight = size.height
    }
    
    //This will be moved to jhScene
    func getArrayOfData() -> NSArray {
        return jhFile.legacyConverterToArray("testdata", "plist")!
    }
    
    func drawDatas() {
        dataLayer = jhLayer(axisDistance, mVerticalRatioToDraw_view, mMargin, mPanelWidth ?? 0, mPanelHeight ?? 0, mFixedPanelWidth, mFixedPanelHeight, layer: 0, panelID: 0)
        
        dataLayer.frame = CGRect(x: 0, y: 0, width: self.mPanelWidth!, height: self.mPanelHeight!) //TODO: will be changed.
        dataLayer.zPosition=1
        //        guideLine.isGeometryFlipped = true
        dataLayer.backgroundColor = UIColor(white: 1, alpha:0.5).cgColor
        self.layer.addSublayer(dataLayer)
        dataLayer.setNeedsDisplay()
        jhData.attachObserver(observer: self)
    }
    
    func jhRedraw() {
        dataLayer.removeFromSuperlayer()
        
        jhData.mCountOfaxes_view = mAllofCountOfDatas
        
        dataLayer = jhLayer(axisDistance, mVerticalRatioToDraw_view, mMargin, mPanelWidth ?? 0, mPanelHeight ?? 0, mFixedPanelWidth, mFixedPanelHeight, layer: 0, panelID: 0)
        
        dataLayer.frame = CGRect(x: 0, y: 0, width: self.mPanelWidth!, height: self.mPanelHeight!) //TODO: will be changed.
        dataLayer.zPosition=1
        //        guideLine.isGeometryFlipped = true
        dataLayer.backgroundColor = UIColor(white: 1, alpha:0.5).cgColor
        self.layer.addSublayer(dataLayer)
        dataLayer.setNeedsDisplay()
        
        drawAxes()
        
    }
    
    func drawEllipse(_ x : CGFloat, _ y : CGFloat, _ width : CGFloat, _ height : CGFloat, thickness : CGFloat, _ color : CGColor){
        //        worldEllipse(context: mContext, getX(x)!, getY(jhDraw.maxR - y)!, width, height, thickness, color)
        if GS.shared.logLevel.contains(.graph) {
            print("worldEllipse(context: mContext,", getX(x+mMargin)!, getY(jhDraw.maxR-y)!, width, height, thickness, color)
        }
        jhDraw.worldEllipse(context: mContext, getX(x+mMargin)!, getY(y)!, width, height, thickness, color)
    }
}

