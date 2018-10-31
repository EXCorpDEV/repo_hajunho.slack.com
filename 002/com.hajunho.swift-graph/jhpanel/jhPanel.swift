//
//  jhPanel.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 15..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

class jhPanel<T> : jhDraw, jhPanel_p, observer_p {
    
    internal var superScene: T?
    //P/ Axes
    let isFixedAxesCount: Bool = true
    let fixedAxesCount: Int = 24
    let mMargin : CGFloat = GV.s.ui_common_margin
    let mLineWidth : CGFloat = GV.s.ui_common_graph_line_width
    
    var jhEnforcingMode: Bool = false
    var jhPanelID: Int = 0
    var dataLayer : CALayer = CALayer(layer: 0)
    var axisLayer : CALayer = CALayer(layer: 0)
    
    internal var mContext : CGContext? = nil
    
    //stored property realted with DATAs
    internal var mCountOfDatas : Int = 0
    
    internal var mColor : CGColor = UIColor.blue.cgColor
    
    /// Axes
    var mUnitOfHorizontalAxes : CGFloat = 100
    var mcountOfHorizontalAxes : Int = 3
    
    
    
    //calculated property related with DATAs' View
    internal var mAllofCountOfDatas : Int {
        get {
            return jhDataCenter.mDatas[0]?.d.count ?? 0
        }
    }
    
    var xDistance: CGFloat {
        //        get {
        //            return (jhDraw.maxR - mMargin * 2) / CGFloat(mAllofCountOfDatas)
        //        }
        
        get {
            return (jhDraw.ARQ  - mMargin * 2) / CGFloat(jhDataCenter.mCountOfdatas_view+1)
        }
        //        set(distance) {
        //            jhDataCenter.mCountOfaxes_view = Int(jhDraw.maxR  / CGFloat(distance))
        //        }
    }
    
    var axisDistance: CGFloat {
        get {
            return (jhDraw.ARQ  - mMargin * 2) / CGFloat(jhDataCenter.mCountOfaxes_view+1)
        }
    }
    
    override init(frame: CGRect) {
        if GS.shared.logLevel.contains(.graphPanel) { print("jhPanel override init(\(frame.width), \(frame.height))")}
        super.init(frame: frame)
        self.layer.isGeometryFlipped = true
        mContext = UIGraphicsGetCurrentContext()
        if GS.shared.logLevel.contains(.graphPanel) { print("jhPanel init color", mLineWidth)}
    }
    
    convenience init(frame: CGRect, scene: inout T?) {
        self.init(frame: frame)
        self.superScene = scene
        if GS.shared.logLevel.contains(.network2) {
            print("ctime in jhPanel = ", (scene as? jhSceneTimeLine)?.currentTime)
            print("ctime in jhPanel = ", (self.superScene as? jhSceneTimeLine)?.currentTime)
        }
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
    
    func getX(_ x: CGFloat) -> CGFloat? {
        var retX : CGFloat? = nil
        retX = x * self.bounds.width / jhDraw.ARQ
        return retX
    }
    
    func getY(_ y: CGFloat) -> CGFloat? {
        var retY : CGFloat? = nil
        retY = y * self.bounds.width / jhDraw.ARQ
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
        drawLine(margin, margin, jhDraw.ARQ-margin, margin)
        drawLine(jhDraw.ARQ-margin, margin, jhDraw.ARQ-margin, jhDraw.ARQ-margin)
        drawLine(jhDraw.ARQ-margin, jhDraw.ARQ-margin, margin, jhDraw.ARQ-margin)
        ////For DEBUG
        //        drawLine(0, 0, jhDraw.maxR, jhDraw.maxR)
        //        drawLine(0, jhDraw.maxR, jhDraw.maxR, 0)
        drawLine(margin, jhDraw.ARQ-margin, margin, margin)
    }
    
    func drawRect(margin : CGFloat, color : CGColor) {
        mColor = color
        drawRect(margin: margin)
    }
    
    /// draw X-axes, Y-axes
    func drawBackboard() {
        if isFixedAxesCount {
            jhDataCenter.mCountOfaxes_view = fixedAxesCount
        } else {
            jhDataCenter.mCountOfaxes_view = mAllofCountOfDatas
        }
        
        jhDataCenter.mCountOfdatas_view = mAllofCountOfDatas
        
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
        
    }
    
    func initDatas() {
        
        var dataSource = getArrayOfData() {
            didSet {
            }
        }
        
        var maxValue : CGFloat = 0.0
        var minValue : CGFloat = jhDraw.ARQ
        
        for element in dataSource {
            let _element = element as! NSArray
            let vDate = _element[0] as! CFDate
            let vNumber = _element[1] as! CGFloat
            
            if GS.shared.logLevel.contains(.graph) {
                print("datasrc2 \(vDate) \(vNumber)")
            }
            
            if vNumber > maxValue { maxValue = vNumber }
            if vNumber < minValue { minValue = maxValue }
            
            jhDataCenter.nonNetworkData.append(vNumber)
        }
        
        GS.shared.testDataMaxValue = maxValue
        GS.shared.testDataMinValue = minValue
        
        GS.shared.testDataVerticalRatioToDraw_view = (jhDraw.ARQ - (2*mMargin)) / GS.shared.testDataMaxValue
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
        
        print("xDistance", xDistance)
        
        dataLayer = jhCommonDataLayer(self, 0)
        
        dataLayer.frame = CGRect(x: 0, y: 0, width: self.bounds.width, height: self.bounds.height) //TODO: will be changed.
        dataLayer.zPosition=1
        //        guideLine.isGeometryFlipped = true
        dataLayer.backgroundColor = UIColor(white: 1, alpha:0.5).cgColor
        self.layer.addSublayer(dataLayer)
        dataLayer.setNeedsDisplay()
        jhDataCenter.attachObserver(observer: self)
    }
    
    func jhRedraw() {
        
        print("xDistance", xDistance)
        
        dataLayer.removeFromSuperlayer()
        
        if isFixedAxesCount {
            jhDataCenter.mCountOfaxes_view = fixedAxesCount
        } else {
            jhDataCenter.mCountOfaxes_view = mAllofCountOfDatas
        }
        
        jhDataCenter.mCountOfdatas_view = mAllofCountOfDatas
        
        print("hjh", xDistance)
        dataLayer = jhCommonDataLayer(self, 0)
        
        dataLayer.frame = CGRect(x: 0, y: 0, width: self.bounds.width, height: self.bounds.height) //TODO: will be changed.
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
            print("worldEllipse(context: mContext,", getX(x+mMargin)!, getY(jhDraw.ARQ-y)!, width, height, thickness, color)
        }
        jhDraw.worldEllipse(context: mContext, getX(x+mMargin)!, getY(y)!, width, height, thickness, color)
    }
}

